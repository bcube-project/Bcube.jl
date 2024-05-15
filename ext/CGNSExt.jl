module CGNSExt
using Bcube
using JLD2

function Bcube.read_file(
    ::Bcube.CGNSIoHandler,
    filepath::String,
    domainNames = String[],
    varnames = String[];
    kwargs...,
)
    println("hello world")
    return 3
end
end