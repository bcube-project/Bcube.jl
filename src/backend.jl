abstract type AbstractBcubeBackend end

struct BcubeBackendCPUSerial <: AbstractBcubeBackend end

#default:
get_bcube_backend() = BcubeBackendCPUSerial()
