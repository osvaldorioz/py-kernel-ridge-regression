#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

namespace py = pybind11;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
typedef Matrix<double, Dynamic, 1> VectorXd;

class KernelRidgeRegression {
private:
    double alpha;
    MatrixXd X_train;
    VectorXd y_train;
    MatrixXd K_inv;

public:
    KernelRidgeRegression(double alpha) : alpha(alpha) {}

    MatrixXd rbf_kernel(const MatrixXd& X1, const MatrixXd& X2, double gamma) {
        int n1 = X1.rows(), n2 = X2.rows();
        MatrixXd K(n1, n2);
        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                K(i, j) = exp(-gamma * (X1.row(i) - X2.row(j)).squaredNorm());
            }
        }
        return K;
    }

    void fit(py::array_t<double> X_np, py::array_t<double> y_np, double gamma) {
        auto X_buf = X_np.request(), y_buf = y_np.request();
        Map<MatrixXd> X(static_cast<double*>(X_buf.ptr), X_buf.shape[0], X_buf.shape[1]);
        Map<VectorXd> y(static_cast<double*>(y_buf.ptr), y_buf.shape[0]);

        X_train = X;
        y_train = y;

        MatrixXd K = rbf_kernel(X, X, gamma);
        K_inv = (K + alpha * MatrixXd::Identity(K.rows(), K.cols())).inverse();
    }

    py::array_t<double> predict(py::array_t<double> X_np, double gamma) {
        auto X_buf = X_np.request();
        Map<MatrixXd> X(static_cast<double*>(X_buf.ptr), X_buf.shape[0], X_buf.shape[1]);

        MatrixXd K_test = rbf_kernel(X, X_train, gamma);
        VectorXd y_pred = K_test * K_inv * y_train;

        return py::array_t<double>(y_pred.size(), y_pred.data());
    }
};

PYBIND11_MODULE(krr_module, m) {
    py::class_<KernelRidgeRegression>(m, "KernelRidgeRegression")
        .def(py::init<double>())
        .def("fit", &KernelRidgeRegression::fit)
        .def("predict", &KernelRidgeRegression::predict);
}
