#include "object.h"

#include <algorithm>
#include <cmath>

Object::Object()
{

}

Object::Object(int label, const std::vector<double> features)
    :label_(label), features_(features)
{

}

bool Object::operator <(const Object &object) const
{
    return label_ < object.label_;
}


