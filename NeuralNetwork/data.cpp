#include "data.h"

#include <fstream>
#include <algorithm>

Data::Data()
{

}

Data::Data(const std::vector<Object> &objects)
    :objects_(objects)
{

}

void Data::add(const Data &data)
{
    objects_.insert(objects_.end(), data.objects_.begin(), data.objects_.end());
}

void Data::add(const Object &object)
{
    objects_.push_back(object);
}

void Data::read(const std::string &filepath, const std::string &filename_labels)
{
    std::ifstream stream(filepath, std::ios_base::binary);
    std::ifstream stream_labels(filename_labels, std::ios_base::binary);

    std::vector<char> images((std::istreambuf_iterator<char>(stream)), (std::istreambuf_iterator<char>()));
    std::vector<char> labels((std::istreambuf_iterator<char>(stream_labels)), (std::istreambuf_iterator<char>()));

    int count = *(int*)(std::vector<char>(images.rend() - 8, images.rend() - 4).data());
    int rows = *(int*)(std::vector<char>(images.rend() - 12, images.rend() - 8).data());;
    int cols = *(int*)(std::vector<char>(images.rend() - 16, images.rend() - 12).data());;

    objects_.resize(count);

#pragma omp parallel for
    for(int i = 0; i < count; ++i)
    {
        std::vector<double> d(rows * cols);

        for(int j = 0; j < rows * cols; ++j)
        {
            d[j] = (unsigned char)images[16 + i * rows * cols + j];
            if (d[j] > 0)
            {
                d[j] = 1;
            }
        }

        objects_[i] = (Object(labels[8 + i], d));
    }

    std::random_shuffle(objects_.begin(), objects_.end());
}

void Data::split_for_test(double part_for_train, Data &train_set, Data &test_set) const
{
    std::vector<Object> objects = objects_;

    std::sort(objects.begin(), objects.end());

    std::vector<Object> train_s;
    std::vector<Object> test_s;

    for(size_t i = 0; i < objects.size();)
    {
        size_t from = i;

        while(i < objects.size() && objects[i].label() == objects[from].label())
        {
            ++i;
        }

        std::random_shuffle(objects.begin() + from, objects.begin() + i);

        for(size_t j = from; j < i; ++j)
        {
            if (j < from + (i - from) * part_for_train)
            {
                train_s.push_back(objects[j]);
            }
            else
            {
                test_s.push_back(objects[j]);
            }
        }
    }

    std::random_shuffle(train_s.begin(), train_s.end());
    std::random_shuffle(test_s.begin(), test_s.end());

    train_set = Data(train_s);
    test_set = Data(test_s);
}

void Data::split(int folder, std::vector<Data> &data) const
{
    std::vector<Object> objects = objects_;

    std::sort(objects.begin(), objects.end());

    std::vector<std::vector<Object>> sets(folder);
    for(size_t i = 0; i < objects.size();)
    {
        size_t from = i;

        while(i < objects.size() && objects[i].label() == objects[from].label())
        {
            ++i;
        }

        std::random_shuffle(objects.begin() + from, objects.begin() + i);

        for(int k = 0; k < folder; ++k)
        {
            for(size_t j = from + (i - from) * k / (folder); j < from + (i - from) * (k + 1) / (folder); ++j)
            {
                sets[k].push_back(objects[j]);
            }
        }
    }

    data.clear();
    for(auto &vec : sets)
    {
        std::random_shuffle(vec.begin(), vec.end());
        data.push_back(Data(vec));
    }
}
