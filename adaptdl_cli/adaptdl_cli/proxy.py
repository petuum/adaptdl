from contextlib import contextmanager
from multiprocessing import Process, Queue

import kubernetes

from mitmproxy.options import Options
from mitmproxy.proxy.config import ProxyConfig
from mitmproxy.proxy.server import ProxyServer
from mitmproxy.tools.dump import DumpMaster
from portpicker import pick_unused_port
from urllib.parse import urlparse


@contextmanager
def service_proxy(namespace, service, listen_host="127.0.0.1", listen_port=0,
                  verbose=False):
    listen_port = listen_port if listen_port else pick_unused_port()
    queue = Queue()
    child = Process(
        target=_run_proxy, name="mitmproxy",
        args=(queue, namespace, service, listen_host, listen_port, verbose),
    )
    child.start()
    queue.get()  # Wait for child to bind the port.
    try:
        yield f"{listen_host}:{listen_port}"
    finally:
        child.terminate()
        child.join()


def _run_proxy(queue, namespace, service, listen_host, listen_port, verbose):
    client = kubernetes.config.new_client_from_config()
    prefix = f"/api/v1/namespaces/{namespace}/services/{service}/proxy"
    options = Options(listen_host=listen_host, listen_port=listen_port,
                      mode=f"reverse:{client.configuration.host}")
    master = DumpMaster(options, with_termlog=verbose, with_dumper=verbose)
    options.keep_host_header = True
    options.ssl_insecure = True
    master.server = ProxyServer(ProxyConfig(options))
    master.addons.add(_Addon(client, prefix))
    queue.put(None)  # Port is bound by this time, unblock parent process.
    master.run()


class _Addon(object):
    def __init__(self, client, prefix):
        self.client = client
        self.prefix = prefix

    def requestheaders(self, flow):
        flow.request.stream = True  # Stream for better performance.
        flow.request.path = self.prefix + flow.request.path
        self.client.update_params_for_auth(  # Inject authorization.
            flow.request.headers, flow.request.query, ["BearerToken"])

    def responseheaders(self, flow):
        flow.response.stream = True  # Stream for better performance.
        if "location" in flow.response.headers:
            # Undo Kubernetes ApiServer's rewriting of the Location header.
            # https://github.com/kubernetes/kubernetes/pull/52556
            url = urlparse(flow.response.headers["location"])
            if url.path.startswith(self.prefix):
                url = url._replace(path=url.path[len(self.prefix):])
            flow.response.headers["location"] = url.geturl()
