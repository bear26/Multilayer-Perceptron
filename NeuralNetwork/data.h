#ifndef DATA
#define DATA

#include "object.h"

#include <vector>
#include <string>

class Data
{
public:
    Data();
    Data(const std::vector<Object> &objects);

    void add(const Data &data);
    void add(const Object &object);

    void read(const std::string &filepath, const std::string &filename_label);

    void split_for_test(double part_for_train, Data &train_set, Data &test_set) const;
    void split(int folder, std::vector<Data> &data) const;

    inline size_t size() const { return objects_.size(); }
    inline Object& operator [](int i) { return objects_[i]; }
    inline const Object& operator [](int i) const { return objects_[i]; }

    inline std::vector<Object>::iterator begin() { return objects_.begin();}
    inline std::vector<Object>::iterator end() { return objects_.end(); }

    inline std::vector<Object>::const_iterator begin() const { return objects_.begin();}
    inline std::vector<Object>::const_iterator end() const { return objects_.end(); }

private:
    std::vector<Object> objects_;
};


#endif // DATA

