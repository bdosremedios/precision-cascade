#ifndef PRECONDITIONER_SPEC_H
#define PRECONDITIONER_SPEC_H

#include <iomanip>
#include <string>
#include <sstream>

struct Preconditioner_Spec {

    std::string name = "";
    double ilutp_tau = -1.0;
    int ilutp_p = -1;

    Preconditioner_Spec() {}

    Preconditioner_Spec(std::string arg_name): name(arg_name) {}

    Preconditioner_Spec(
        std::string arg_name, double arg_ilutp_tau, int arg_ilutp_p
    ):
        name(arg_name), ilutp_tau(arg_ilutp_tau), ilutp_p(arg_ilutp_p)
    {}

    bool is_default() const {
        return ((name == "") && (ilutp_tau == -1.0) && (ilutp_p == -1));
    }

    std::string get_spec_string() const {
        if ((ilutp_tau == -1.0) && (ilutp_p == -1)) {
            return name + "_NA_NA";
        } else {
            std::stringstream ilutp_tau_strm;
            ilutp_tau_strm << std::setprecision(3);
            ilutp_tau_strm << ilutp_tau;
            std::string ilutp_tau_str = ilutp_tau_strm.str();
            for (int i=0; i<ilutp_tau_str.size(); ++i) {
                if (ilutp_tau_str[i] == '.') {
                    ilutp_tau_str.erase(i, 1);
                    --i;
                }
            }
            return name + "_" + ilutp_tau_str + "_" + std::to_string(ilutp_p);
        }
    }

    Preconditioner_Spec(const Preconditioner_Spec &other) {
        *this = other;
    }

    Preconditioner_Spec &operator=(
        const Preconditioner_Spec &other
    ) {

        name = other.name;
        ilutp_tau = other.ilutp_tau;
        ilutp_p = other.ilutp_p;

        return *this;

    }

    bool operator==(const Preconditioner_Spec &other) const {
        return (
            (name == other.name) &&
            (ilutp_tau == other.ilutp_tau) &&
            (ilutp_p == other.ilutp_p)
        );
    }

};

#endif