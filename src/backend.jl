# abstract type AbstractBcubeBackend{T} end

# struct BcubeBackend{T} <: AbstractBcubeBackend{T}
#     backend::T
# end

struct BcubeCPUSerial end

#default:
get_bcube_backend() = BcubeCPUSerial()

abstract type AbstractBcubeBackendStyle end

struct CPUSerialStyle <: AbstractBcubeBackendStyle end
backend_style(::BcubeCPUSerial) = CPUSerialStyle()

abstract type AbstractAtomicStyle end
struct AtomicNeeded <: AbstractAtomicStyle end
struct AtomicNotNeeded <: AbstractAtomicStyle end

#default:
isAtomicNeeded(::AbstractBcubeBackendStyle) = AtomicNeeded()
isAtomicNeeded(::CPUSerialStyle) = AtomicNotNeeded()