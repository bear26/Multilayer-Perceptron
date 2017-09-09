#include "model.h"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <numeric>
#include <cassert>
#include <fstream>

const int count_class = 10;
const std::vector<int> count_in_layer = {784, 500, 150, 10};

Model::Model()
{

}

void Model::train(const Data &data, const std::string &snapshot)
{
	const int count_samples = 60000;
	const int number_of_epochs = 200;
	// count samples * number of epochs
    const int count_train_step = count_samples * number_of_epochs;

    for(size_t i = 0; i < count_in_layer.size() - 1; ++i)
    {
        weight_.push_back(std::vector<std::vector<double>>(count_in_layer[i], std::vector<double>(count_in_layer[i + 1])));

        for(auto &v : weight_.back())
        {
            std::generate(v.begin(), v.end(), [](){ return ((double)rand()) / RAND_MAX / 10 - 0.05; });
        }
    }

    for(int v : count_in_layer)
    {
        out_.push_back(std::vector<double>(v));

        w0_.push_back(std::vector<double>(v));
        std::generate(w0_.back().begin(), w0_.back().end(), [](){ return ((double)rand()) / RAND_MAX / 100 - 0.05; });
    }

    for(int i = 0; i < count_train_step; ++i)
    {
        learn_on_object(data[i % data.size()]);

        if (i % 100 == 0)
        {
            int correct = 0;
            for(int j = 0; j < 100; ++j)
            {
                correct += predict(data[j]) == data[j].label();
            }

            std::cout << correct << " " << i / count_samples << std::endl;
        }

        if (i && i % (count_samples) == 0)
        {
            std::cout << "SAVE" << std::endl;
            save(snapshot + std::to_string(i / count_samples));
        }
    }
}

int Model::predict(const Object &object)
{
    predict_out(object.features());

    return std::distance(out_.back().begin(), std::max_element(out_.back().begin(), out_.back().end()));
}

std::vector<std::pair<int, int> > Model::test(const Data &data)
{
    std::vector<std::pair<int, int>> ans(data.size());

//#pragma omp parallel for
    for(size_t i = 0; i < data.size(); ++i)
    {
        ans[i] = (std::make_pair(data[i].label(), predict(data[i])));
    }

    return ans;
}

void Model::save(const std::string &filename) const
{
    std::fstream stream(filename, std::ios_base::out);

    stream << weight_.size() << std::endl;
    for(auto v : weight_)
    {
        stream << v.size() << std::endl;
        for(auto v1 : v)
        {
            stream << v1.size() << std::endl;
            for(auto v2 : v1)
            {
                stream << v2 << " ";
            }
            stream << std::endl;
        }
        stream << std::endl;
    }

    stream << w0_.size() << std::endl;
    for(auto v : w0_)
    {
        stream << v.size() << std::endl;
        for(auto v1 : v)
        {
            stream << v1 << " ";
        }
        stream << std::endl;
    }
}

void Model::load(const std::string &filename)
{
    std::fstream stream(filename);

    for(int v : count_in_layer)
    {
        out_.push_back(std::vector<double>(v));
    }
    int n;
    stream >> n;
    weight_.resize(n);

    for(size_t i = 0; i < weight_.size(); ++i)
    {
        stream >> n;
        weight_[i].resize(n);

        for(size_t j = 0; j < weight_[i].size(); ++j)
        {
            stream >> n;
            weight_[i][j].resize(n);
            for(size_t k = 0; k < weight_[i][j].size(); ++k)
            {
                stream >> weight_[i][j][k];
            }
        }
    }

    stream >> n;
    w0_.resize(n);
    for(size_t i = 0; i < w0_.size(); ++i)
    {
        stream >> n;
        w0_[i].resize(n);
        for(size_t j = 0; j < w0_[i].size(); ++j)
        {
            stream >> w0_[i][j];
        }
    }
}

void Model::learn_on_object(const Object &obj)
{
    std::vector<double> out(out_.back().size(), 0);
    out[obj.label()] = 1;

    predict_out(obj.features());

    update_weight(out);
}

void Model::predict_out(const std::vector<double> &f)
{
    assert(f.size() == out_[0].size());

    for(size_t i = 0; i < f.size(); ++i)
    {
        out_[0][i] = f[i];
    }

    for(size_t i = 1; i < out_.size(); ++i)
    {
#pragma omp parallel for
        for(int j = 0; j < (int)out_[i].size(); ++j)
        {
            out_[i][j] = w0_[i][j];

            for(size_t k = 0; k < out_[i - 1].size(); ++k)
            {
                out_[i][j] += out_[i - 1][k] * weight_[i - 1][k][j];
            }

            out_[i][j] = func_activation(out_[i][j]);
        }
    }
}

void Model::update_weight(const std::vector<double> &ans)
{
    const double step = 0.001;
    std::vector<std::vector<double>> delta;

    std::vector<double> d_last(out_.back().size());

    for(size_t i = 0; i < out_.back().size(); ++i)
    {
        double out = out_.back()[i];
        d_last[i] = 2 * out * (1 - out) * (ans[i] - out);
    }

    delta.insert(delta.begin(), d_last);

    for(int i = out_.size() - 2; i >= 0; --i)
    {
        std::vector<double> d(out_[i].size());

#pragma omp parallel for
        for(int j = 0; j < (int)out_[i].size(); ++j)
        {
            double out = out_[i][j];
            double sum = 0;

            for(size_t k = 0; k < out_[i + 1].size(); ++k)
            {
                sum += delta[0][k] * weight_[i][j][k];
            }

            d[j] = 2 * out * (1 - out) * sum;
        }

        delta.insert(delta.begin(), d);
    }

    for(size_t i = 0; i < weight_.size(); ++i)
    {
        for(size_t j = 0; j < weight_[i].size(); ++j)
        {
            for(size_t k = 0; k < weight_[i][j].size(); ++k)
            {
                weight_[i][j][k] += step * out_[i][j] * delta[i + 1][k];
            }
        }
    }

    for(size_t i = 0; i < weight_.size(); ++i)
    {
        for(size_t j = 0; j < weight_[i].size(); ++j)
        {
            w0_[i][j] += step * delta[i][j];
        }
    }
}

double Model::func_activation(double v) const
{
    return 1 / (1 + std::exp(-2 * v));
}

double print_result(const std::vector<std::pair<int, int> > &result, bool print_confusion_matrix)
{
    std::vector<std::vector<int>> conf_matrix(count_class, std::vector<int>(count_class, 0));

	std::vector<int> count_samples(count_class);

    for(auto pair : result)
    {
        conf_matrix[pair.first][pair.second]++;
		count_samples[pair.first]++;
    }

    for(size_t i = 0; i < conf_matrix.size(); ++i)
    {
        double presision = 0;
        double recall = 0;

        for(size_t j = 0; j < conf_matrix[i].size(); ++j)
        {
            presision += conf_matrix[i][j];
        }

        for(size_t j = 0; j < conf_matrix.size(); ++j)
        {
            recall += conf_matrix[j][i];
        }

        presision = conf_matrix[i][i] / presision;
        recall = conf_matrix[i][i] / recall;
    }

    int correct = 0;
    for(size_t i = 0; i < conf_matrix.size(); ++i)
    {
        correct += conf_matrix[i][i];
    }

	if (print_confusion_matrix)
	{
		for (int label = 0; label < count_class; ++label)
		{
			for (int acc : conf_matrix[label])
			{
				printf("%.3lf ", 1.0 * acc / count_samples[label]);
			}
			printf("\n");
		}
	}
	else
	{
		printf("%lf", 100.0 * correct / result.size());
	}

    std::cout.flush();

    return 100.0 * correct / result.size();
}
