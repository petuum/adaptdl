from contextlib import contextmanager
from multiprocessing import Process
from subprocess import Popen, DEVNULL

from mitmproxy.options import Options
from mitmproxy.proxy.config import ProxyConfig
from mitmproxy.proxy.server import ProxyServer
from mitmproxy.tools.dump import DumpMaster
from portpicker import pick_unused_port
from urllib.parse import urlparse


@contextmanager
def service_proxy(namespace, name, host="127.0.0.1", port=0):
    port = port if port else pick_unused_port()
    prefix = "/api/v1/namespaces/{}/services/{}/proxy".format(namespace, name)
    kproxy_port = pick_unused_port()
    kproxy_args = ["kubectl", "proxy", f"--port={kproxy_port}",
                   "--accept-hosts=.*", f"--api-prefix={prefix}"]
    with Popen(kproxy_args, stdout=DEVNULL) as kproxy:
        try:
            options = Options(listen_host=host, listen_port=port,
                              mode=f"reverse:http://localhost:{kproxy_port}")
            mproxy = DumpMaster(options, with_termlog=False, with_dumper=False)
            options.keep_host_header = True
            mproxy.server = ProxyServer(ProxyConfig(options))
            mproxy.addons.add(_Middleware(prefix))
            p = Process(target=mproxy.run)
            p.start()
            try:
                yield f"{host}:{port}"
            finally:
                p.terminate()
                p.join()
        finally:
            kproxy.kill()


class _Middleware(object):
    def __init__(self, prefix):
        self.prefix = prefix

    def request(self, flow):
        flow.request.path = self.prefix + flow.request.path

    def response(self, flow):
        if "location" in flow.response.headers:
            url = urlparse(flow.response.headers["location"])
            if url.path.startswith(self.prefix):
                url = url._replace(path=url.path[len(self.prefix):])
            flow.response.headers["location"] = url.geturl()
