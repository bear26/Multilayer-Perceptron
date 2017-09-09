#ifndef MODEL
#define MODEL

#include <memory>

#include "data.h"

class Model
{
public:
    Model();

    void train(const Data &data, const std::string &snapshot);

    void learn_on_object(const Object &obj);

    int predict(const Object &object);

    //return vector pair(real label, prediction label)
    std::vector<std::pair<int, int>> test(const Data &data);

    void save(const std::string &filename) const;
    void load(const std::string &filename);

private:
    std::vector<std::vector<std::vector<double>>> weight_;
    std::vector<std::vector<double>> out_;
    std::vector<std::vector<double>> w0_;

    void predict_out(const std::vector<double> &f);
    void update_weight(const std::vector<double> &ans);
    double func_activation(double v) const;
};

double print_result(const std::vector<std::pair<int, int>> &result, bool print_confusion_matrix);


#endif // MODEL

