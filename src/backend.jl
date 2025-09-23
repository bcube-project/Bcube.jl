"""
Abstract structure intended to wrap KernelAbstractions to avoid having it as a dependence
"""
abstract type AbstractBcubeBackend end

"""
Default Bcube CPU backend
"""
struct BcubeBackendCPUSerial <: AbstractBcubeBackend end

#default:
get_bcube_backend() = BcubeBackendCPUSerial()
