#ifndef TORC_EXPLICIT_FN_H
#define TORC_EXPLICIT_FN_H

namespace torc::fn {
    /**
     * Class implementation of an explicit function, where the differentials are provided by the user.
     * @tparam scalar_t the type of scalar used for the fn
     */
    template <class scalar_t>
    class ExplicitFn {
        using matrixx_t = Eigen::MatrixX<scalar_t>;
        using vectorx_t = Eigen::VectorX<scalar_t>;

    public:
        ExplicitFn() {
            const size_t dim = 1;
            this->func_ = [](const vectorx_t& x) {return scalar_t(0);};
            this->grad_ = [](const vectorx_t& x) {return vectorx_t::Zero(dim);};
            this->hess_ = [](const vectorx_t& x) {return matrixx_t::Zero(dim, dim);};
        };

        /**
         * Constructor for the ExplicitFn class
         * @param func the func function
         * @param grad the gradient of the function
         * @param hess the hessian of the function
         * @param dim the dimension of the function
         * @param fn_name the name of the function
         */
        ExplicitFn(const std::function<scalar_t(vectorx_t)>& func,
                   const std::function<vectorx_t(vectorx_t)>& grad,
                   const std::function<matrixx_t(vectorx_t)>& hess,
                   const size_t& dim=1,
                   const std::string& fn_name="ExplicitFnInstance") {
            if (dim <= 0) {
                throw std::runtime_error("Dimension must be greater than 1.");
            }
            this->dim_ = dim;
            this->SetName(fn_name);
            this->func_ = func;
            this->grad_ = grad;
            this->hess_ = hess;
        }

        /**
         * Evaluates the function at a point
         * @param x the input
         * @return f(x)
         */
        scalar_t Evaluate(const vectorx_t& x) const { return this->func_(x); }

        /**
         * Evaluates the function at a point
         * @param x the input
         * @return f(x)
         */
        scalar_t operator() (const vectorx_t& x) const { return this->func_(x); }

        /**
         * Adds two functions. Taking derivatives is a linear operation, so the gradients and the hessians are added.
         * @param other the other function
         * @return the sum of the two functions
         */
        ExplicitFn<scalar_t> operator+ (const ExplicitFn<scalar_t>& other) const {
            if (this->dim_ != other.dim_) {
                throw std::runtime_error("Two functions must have the same dimension.");
            }
            ExplicitFn<scalar_t> fn {};
            fn.func_ = [this, other](const vectorx_t& x) { return this->func_(x) + other.func_(x); };
            fn.grad_ = [this, other](const vectorx_t& x) {
                const size_t dim = this->dim_;
                vectorx_t grad(dim);
                vectorx_t grad1 = this->grad_(x);
                vectorx_t grad2 = other.grad_(x);
                for (int i=0; i<dim; i++) {
                    grad(i) = grad1(i) + grad2(i);  // add manually to prevent bad_alloc from adding two dynamic sized
                }
                return grad;
            };
            fn.hess_ = [this, other](const vectorx_t& x) {
                const size_t dim = this->dim_;
                matrixx_t hess(dim, dim);
                matrixx_t hess1 = this->hess_(x);
                matrixx_t hess2 = other.hess_(x);
                for (int i=0; i<dim; i++) {
                    for (int j=0; j<dim; j++) {
                        hess(i, j) = hess1(i, j) + hess2(i, j);
                    }
                }
                return hess;
            };
            fn.dim_ = this->dim_;
            fn.name_ = this->name_ + "_sum_" + other.name_;
            return fn;
        }

        /**
         * Subtracts two functions. Differentiation is linear, so the derivatives are subtracted.
         * @param other the other function
         * @return the difference of the two functions
         */
        ExplicitFn<scalar_t> operator- (const ExplicitFn<scalar_t>& other) const {
            if (this->dim_ != other.dim_) {
                throw std::runtime_error("Two functions must have the same dimension.");
            }
            ExplicitFn<scalar_t> fn {};
            fn.func_ = [this, other](const vectorx_t& x) { return this->func_(x) - other.func_(x); };
            fn.grad_ = [this, other](const vectorx_t& x) {
                const size_t dim = this->dim_;
                vectorx_t grad(dim);
                vectorx_t grad1 = this->grad_(x);
                vectorx_t grad2 = other.grad_(x);
                for (int i=0; i<dim; i++) {
                    grad(i) = grad1(i) - grad2(i);  // add manually to prevent bad_alloc from adding two dynamic sized
                }
                return grad;
            };
            fn.hess_ = [this, other](const vectorx_t& x) {
                const size_t dim = this->dim_;
                matrixx_t hess(dim, dim);
                matrixx_t hess1 = this->hess_(x);
                matrixx_t hess2 = other.hess_(x);
                for (int i=0; i<dim; i++) {
                    for (int j=0; j<dim; j++) {
                        hess(i, j) = hess1(i, j) - hess2(i, j);
                    }
                }
                return hess;
            };
            fn.dim_ = this->dim_;
            fn.name_ = this->name_ + "_sum_" + other.name_;
            return fn;
        }

        /**
         * Evaluates the gradient of the function at a point
         * @param x the input
         * @return grad f(x)
         */
        vectorx_t Gradient(const vectorx_t& x) const { return this->grad_(x); }

        /**
         * Evaluates the Hessian of the function at a point
         * @param x the input
         * @return H_f(x)
         */
        matrixx_t Hessian(const vectorx_t& x) const {
            return this->hess_(x);
        }

        /**
         * Returns the identifier_ of the function
         * @return the function's name
         */
        [[nodiscard]] std::string GetName() const { return this->name_; }

        /**
         * Returns the domain's dimension of the function
         * @return the domain's dimension
         */
        [[nodiscard]] size_t GetDim() const { return this->dim_; }


    protected:
        std::function<scalar_t(vectorx_t)> func_;   // the original function
        std::function<vectorx_t(vectorx_t)> grad_;  // the gradient of the function
        std::function<matrixx_t(vectorx_t)> hess_;  // the hessian of the function
        size_t dim_ = 1;
        std::string name_ = "ExplicitFnInstance";

        /**
         * Setter for the identifier attribute. Checks whether the string given is a valid variable name. This function
         * is intended for internal use in subclasses only, since changing the name of a function after it has been
         * loaded into dynamic libraries cause complications for later loading.
         * @param str the identifier
         */
        void SetName(const std::string &str) {
            if (!IsValidName(str)) {
                throw std::runtime_error("Identifier must be a valid variable name.");
            }
            this->name_ = str;
        }

        /**
         * Checks whether a string is a valid identifier (i.e., starts with a alphabetical character and contains only
         * alpha-numerical characters and underscores).
         * @param str the string to check
         * @return true if the string is a valid identifier, false otherwise
         */
        static bool IsValidName(const std::string &str) {
            if (!isalpha(str[0]) && str[0] != '_') {
                return false;
            }
            return std::all_of(str.cbegin(), str.cend(), [](char c) {return (isalnum(c) || c=='_');});
        }
    };
}
#endif //TORC_EXPLICIT_FN_H
