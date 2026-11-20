class ServiceError(Exception):
    """An error whose public code never includes upstream data or credentials."""

    code = "service_error"
    status = 503


class ContractError(ServiceError):
    code = "contract_error"
    status = 502


class NotConfigured(ServiceError):
    code = "not_configured"


class UpstreamError(ServiceError):
    code = "upstream_error"
    status = 502


class UpstreamAuthenticationError(UpstreamError):
    code = "upstream_authentication_failed"


class ResourceExhausted(ServiceError):
    code = "resource_exhausted"


class InvalidModelOutput(ServiceError):
    code = "invalid_model_output"


class OfflineMode(ServiceError):
    code = "network_disabled"


class TransientLLMError(ServiceError):
    code = "llm_transient_failure"

    def __init__(self, retry_after: float = 5.0):
        super().__init__(self.code)
        self.retry_after = retry_after
