#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>

#include "data.h"
#include "model.h"

Model model_predict;

int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage <train|test|testEx> <dataset_im> <dataset_labels> <weight for test>" << std::endl;
        return -1;
    }

    srand(time(nullptr));

    if (std::string(argv[1]) == "train")
    {
        Data data;
        data.read(argv[2], argv[3]);

        Model model;

        model.train(data, argv[4]);

        model.save(argv[4]);
    }
    else if (std::string(argv[1]) == "test")
    {
        Data data;
        data.read(argv[2], argv[3]);

        Model model;

        model.load(argv[4]);

        auto result = model.test(data);

        print_result(result, false);
    }
	else if (std::string(argv[1]) == "testEx")
	{
		Data data;
		data.read(argv[2], argv[3]);

		Model model;

		model.load(argv[4]);

		auto result = model.test(data);

		print_result(result, true);
	}

    return 0;
}

