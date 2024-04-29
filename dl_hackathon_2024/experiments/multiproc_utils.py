# Copyright (C) 2023 Langlois Quentin, ICTEAM, UCLouvain. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import multiprocessing

logger = logging.getLogger(__name__)

def run_in_processes(fn, args, max_workers = multiprocessing.cpu_count(), tqdm = lambda x: x):
    """ Iterates over `args` to create processes that run `fn(args[i])`, with at most `max_workers` running in parallel """
    buffer  = multiprocessing.Queue()
    workers = [
        multiprocessing.Process(target = _run_in_process, args = (fn, buffer, i, arg))
        for i, arg in enumerate(args)
    ]

    worker_idx = min(len(workers), max(1, max_workers))
    logger.debug('Starting {}/{} initial workers'.format(worker_idx, len(workers)))
    for i in range(worker_idx): workers[i].start()

    results = [None] * len(workers)
    try:
        for i in tqdm(range(len(workers))):
            idx, res = buffer.get()
            results[idx] = res
            if worker_idx < len(workers):
                logger.debug('Start worker #{}/{}'.format(worker_idx + 1, len(workers)))
                workers[worker_idx].start()
                worker_idx += 1
    
        for w in workers: w.join()
    finally:
        for w in workers: w.terminate()
    
    return results

def _run_in_process(fn, queue, idx, args):
    try:
        res = fn(args)
    except Exception as e:
        res = e
    finally:
        queue.put((idx, res))

