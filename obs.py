from langfuse import Langfuse
from config import Config

class Tracer:
  """Safe Langfuse wrapper that falls back to no-op if unconfigured."""
  
  def __init__(self, config: Config):
    self.enabled = bool(config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY)
    if self.enabled:
      self.client = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST
      )
    else:
      self.client = None

  def create_trace(self, name: str, input_data: dict):
    if not self.enabled:
      return DummyTrace()
    return self.client.trace(name=name, input=input_data)

  def flush(self):
    if self.enabled and self.client:
      self.client.flush()


class DummyTrace:
  """No-op trace object when Langfuse is not enabled."""
  def generation(self, **kwargs):
    return DummySpan()
  
  def span(self, **kwargs):
    return DummySpan()
  
  def update(self, **kwargs):
    pass


class DummySpan:
  """No-op span object."""
  def end(self, **kwargs):
    pass
  
  def update(self, **kwargs):
    pass