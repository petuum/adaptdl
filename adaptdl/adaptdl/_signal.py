# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import signal


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_exit_flag():
    return EXIT_FLAG


def _handler(signum, frame):
    global EXIT_FLAG
    EXIT_FLAG = True
    LOG.debug("Got signal {}...".format(signum))
    if signum == signal.SIGINT:
        LOG.info("Got SIGINT, exiting gracefully... "
                 "Send signal again to force exit.")
        signal.signal(signal.SIGINT, SIGINT_HANDLER)


EXIT_FLAG = False
SIGINT_HANDLER = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGTERM, _handler)
signal.signal(signal.SIGINT, _handler)
