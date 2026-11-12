using Paulimorphic
using ITensors
using ITensorMPS
using ITensorMPOConstruction

"""
    paulisum_to_opsum(H::Paulimorphic.PauliSum)

Convert a Paulimorphic PauliSum into an ITensorMPS OpSum.

Map each Pauli-string position to the corresponding qubit site.
Omit identity factors within each string, and represent a pure
identity term using "I" on site 1.

Preserve the coefficients without applying truncation.
"""

function paulisum_to_opsum(H::Paulimorphic.PauliSum)
    n = Paulimorphic.countSites(H)
    @assert n >= 1. "The Hamiltonian must contain at least one qubit site."

    os = ITensorMPS.OpSum()

    for (pauli_string, coefficient) in Paulimorphic.collectTerms(H)
        symbols = Paulimorphic.toString(
            pauli_string;
            denseString = true
        )

        @assert length(symbols) == n "Pauli-string length does not match the number of sites."

        operators = Any[]

        for (j, symbol) in enumerate(symbols)
            if symbol == 'I'
                continue
            end

            @assert symbol in ('X', 'Y', 'Z') "Unsupported Pauli symbol: $symbol"
            push!(operators, string(symbol), j)
        end

        if isempty(operators)
            ITensorMPS.add!(os, coefficient, "I", 1)
        else
            ITensorMPS.add!(os, coefficient, operators...)
        end
    end

    return os
end