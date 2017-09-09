#ifndef OBJECT
#define OBJECT

#include <vector>

class Object
{
public:
    Object();
    Object(int label, const std::vector<double> features);

    inline int label() const { return label_; }
    inline const std::vector<double> &features() const { return features_; }

    bool operator <(const Object &object) const;

private:
    int label_;
    std::vector<double> features_;
};

#endif // OBJECT

