"""
    _compute_space_dim(topodim, lx, ly, lz, tol, verbose::Bool)

Deduce the number of space dimensions from the mesh boundaries : if one (or more) dimension of the bounding
box is way lower than the other dimensions, the number of space dimension is decreased.

Currently, having for instance (x,z) is not supported. Only (x), or (x,y), or (x,y,z).
"""
function _compute_space_dim(topodim, lx, ly, lz, tol, verbose::Bool)

    # Maximum dimension and default value for number of space dimensions
    lmax = maximum([lx, ly, lz])

    # Now checking several case (complex `if` cascade for comprehensive warning messages)

    # If the topology is 3D, useless to continue, the space dim must be 3
    topodim == 3 && (return 3)

    if topodim == 2
        if (lz / lmax < tol)
            msg = "Warning : the mesh is flat on the z-axis. It is now considered 2D."
            msg *= " (use `spacedim` argument of `read_msh` if you want to keep 3D coordinates.)"
            msg *= " Disable this warning with `verbose = false`"
            verbose && println(msg)

            return 2
        else
            # otherwise, it is a surface in a 3D space, we keep 3 coordinates
            return 3
        end
    end

    if topodim == 1
        if (ly / lmax < tol && lz / lmax < tol)
            msg = "Warning : the mesh is flat on the y and z axis. It is now considered 1D."
            msg *= " (use `spacedim` argument of `read_msh` if you want to keep 2D or 3D coordinates.)"
            msg *= " Disable this warning with `verbose = false`"
            verbose && println(msg)

            return 1
        elseif (ly / lmax < tol && lz / lmax > tol)
            error(
                "You have a flat y-axis but a non flat z-axis, this is not supported. Consider rotating your mesh.",
            )

        elseif (ly / lmax > tol && lz / lmax < tol)
            msg = "Warning : the mesh is flat on the z-axis. It is now considered as a 1D mesh in a 2D space."
            msg *= " (use `spacedim` argument of `read_msh` if you want to keep 3D coordinates.)"
            msg *= " Disable this warning with `verbose = false`"
            verbose && println(msg)

            return 2
        else
            # otherwise, it is a line in a 3D space, we keep 3 coordinates
            return 3
        end
    end
end