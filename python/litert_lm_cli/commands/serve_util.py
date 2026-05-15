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

"""Shared core utilities for managing LiteRT-LM serving lifecycles."""

from __future__ import annotations

import http.server

import click

import litert_lm
from litert_lm_cli import model


class LiteRTLMServer(http.server.HTTPServer):
  """Custom HTTP server tracking persistent LiteRT-LM engine lifecycles."""

  def __init__(
      self,
      server_address: tuple[str, int],
      RequestHandlerClass: type[http.server.BaseHTTPRequestHandler],
  ):
    super().__init__(server_address, RequestHandlerClass)
    self.litert_lm_engine: litert_lm.Engine | None = None
    self.model_id: str | None = None


def get_or_initialize_server_engine(
    server: LiteRTLMServer, model_id: str
) -> litert_lm.Engine:
  """Retrieves the persistent server engine or initializes it on first request.

  Lifetime Management:
  The LiteRT-LM Engine is a globally scoped persistent resource attached
  directly to explicit runtime properties on the custom server context object.
  - Initialization: Invokes `__enter__` dynamically upon the arrival of the
    first incoming inference request.
  - Termination: The running server's parent execution process is responsible
    for explicitly invoking `__exit__` on `server.litert_lm_engine` during outer
    context teardown loops (e.g., in `run_server` finally blocks).

  Args:
    server: The active custom LiteRTLMServer instance object.
    model_id: The requested model identifier string.

  Returns:
    The shared LiteRT-LM Engine context object.

  Raises:
    FileNotFoundError: If the model package path does not exist.
    RuntimeError: If a different model ID is requested after initialization.
  """
  if server.litert_lm_engine is not None:
    # TODO: b/513076049 - support multiple engines.
    if server.model_id != model_id:
      raise RuntimeError(
          f"Server already initialized with model {server.model_id!r}. "
          f"Switching to {model_id!r} is not supported."
      )
    return server.litert_lm_engine

  m = model.Model.from_model_id(model_id)
  if not m.exists():
    raise FileNotFoundError(f"Model {model_id} not found")

  click.echo(
      click.style(f"Initializing engine for model: {m.model_path}", fg="cyan")
  )
  engine = litert_lm.Engine(m.model_path, backend=litert_lm.Backend.CPU())
  engine.__enter__()
  server.litert_lm_engine = engine
  server.model_id = model_id
  return engine
