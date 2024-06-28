#ifndef TYPEIDENTITY_H
#define TYPEIDENTITY_H

namespace cascade {

// solution from https://stackoverflow.com/questions/3052579/explicit-
// specialization-in-non-namespace-scope for explicit specialization in
// non-namespace scope
template <template <typename> typename TMatrix>
struct TMatrixIdentity { typedef TMatrix<double> _; };

template <typename TPrecision>
struct TypeIdentity { typedef TPrecision _; };

}

#endif