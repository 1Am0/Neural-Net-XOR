#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>

std::mt19937 rng(std::random_device{}());

struct Sample
{
	std::vector<double> in;
	int label;
};

struct Neuron
{
	std::vector<double> w;
	double b = 0;
};

struct ForwardResult
{
	std::vector<double> h;
	double p;
};

struct Model
{
	std::vector<Neuron> hidden;
	Neuron output;
};

double sigmoid(double z)
{
	return 1 / (1 + std::exp(-z));
}

std::vector<Sample> createSample(int n)
{
	std::uniform_real_distribution<double> val(-1, 1);

	std::vector<Sample> samples;

	samples.reserve(n);

	for (int i = 0; i < n; i++)
	{
		double in1 = val(rng);
		double in2 = val(rng);
		int label = (in1 * in2) >= 0 ? 1 : 0;

		samples.push_back({ { in1, in2 }, label });
	}

	return samples;
}

void initNeuron(Neuron& n, int numInputs)
{
	n.w.resize(numInputs);

	std::uniform_real_distribution<double> randWeight(-0.5, 0.5);

	for (double& weight : n.w)
	{
		weight = randWeight(rng);
	}

	n.b = randWeight(rng);
}

void initModel(Model& m)
{
	m.hidden.resize(4);

	for (Neuron& n : m.hidden)
	{
		initNeuron(n, 2);
	}

	initNeuron(m.output, m.hidden.size());
}

ForwardResult forward(const Model& m, const Sample& s)
{
	std::vector<double> h;
	for (const Neuron& n : m.hidden)
	{
		double val = n.b;
		for (int i = 0; i < n.w.size(); i++)
		{
			val += n.w[i] * s.in[i];
		}

		double z = sigmoid(val);

		h.push_back(z);
	}

	const Neuron& n = m.output;

	double val = n.b;
	for (int i = 0; i < h.size(); i++)
	{
		val += n.w[i] * h[i];
	}

	double z = sigmoid(val);

	return { h, z };
}

double computeLoss(const Model& m, const std::vector<Sample>& samples)
{
	double tLoss = 0;

	for (const Sample& s : samples)
	{
		double p = forward(m, s).p;
		p = std::clamp(p, 0.0000001, 1 - 0.0000001);

		double y = s.label;

		tLoss += -(y * std::log(p) + (1 - y) * (std::log(1 - p)));
	}

	return tLoss / samples.size();
}

void zeroNeuron(Neuron& n)
{
	for (double& weight : n.w)
	{
		weight = 0;
	}
	n.b = 0;
}

Model zeroModel(Model m)
{
	for (Neuron& n : m.hidden)
	{
		zeroNeuron(n);
	}
	zeroNeuron(m.output);

	return m;
}

Model computeGradient(const Model& m, const std::vector<Sample>& samples)
{
	Model g = zeroModel(m);

	for (const Sample& s : samples)
	{
		auto [h, p] = forward(m, s);

		double error = (p - s.label) / samples.size();

		for (int i = 0; i < h.size(); i++)
		{
			g.output.w[i] += error * h[i];
		}
		g.output.b += error;
		
		for (int i = 0; i < h.size(); i++)
		{
			Neuron& n = g.hidden[i];

			double eJ = error * m.output.w[i] * h[i] * (1 - h[i]);

			for (int j = 0; j < s.in.size(); j++)
			{
				n.w[j] += eJ * s.in[j];
			}
			n.b += eJ;
		}
	}

	return g;
}

void train(Model& m, const std::vector<Sample>& sample, int epochs, double lr)
{
	for (int e = 0; e < epochs; e++)
	{
		Model g = computeGradient(m, sample);

		for (int i = 0; i < m.hidden.size(); i++)
		{
			Neuron& n = m.hidden[i];
			Neuron& added = g.hidden[i];

			for (int j = 0; j < n.w.size(); j++)
			{
				n.w[j] -= added.w[j] * lr;
			}

			n.b -= added.b * lr;
		}

		Neuron& n = m.output;
		Neuron& added = g.output;

		for (int j = 0; j < n.w.size(); j++)
		{
			n.w[j] -= added.w[j] * lr;
		}

		n.b -= added.b * lr;
	}
}

double accuracy(const Model& m, const std::vector<Sample>& samples)
{
	double c = 0;

	for (const Sample& s : samples)
	{
		int p = (forward(m, s).p >= 0.5) ? 1 : 0;
		double e = (p == s.label) ? 1 : 0;

		c += e;
	}

	return c / samples.size() * 100;
}

// Writes nn_output.csv for the visualizer. Columns: kind,x1,x2,label,p
//   grid: p for a gridN x gridN grid over [-1,1]^2 (label unused = -1)
//   test: each test sample with its true label and p
//   line: one row per hidden neuron: x1 = w[0], x2 = w[1], label = b, p = output weight
void exportCSV(const Model& m, const std::vector<Sample>& test, const std::string& path, int gridN = 60)
{
	std::ofstream f(path);
	f << "kind,x1,x2,label,p\n";

	for (int r = 0; r < gridN; r++)
	{
		for (int c = 0; c < gridN; c++)
		{
			double x1 = -1 + 2.0 * c / (gridN - 1);
			double x2 = -1 + 2.0 * r / (gridN - 1);
			f << "grid," << x1 << "," << x2 << ",-1," << forward(m, { { x1, x2 }, -1 }).p << "\n";
		}
	}

	for (const Sample& s : test)
	{
		f << "test," << s.in[0] << "," << s.in[1] << "," << s.label << "," << forward(m, s).p << "\n";
	}

	for (size_t j = 0; j < m.hidden.size(); j++)
	{
		const Neuron& n = m.hidden[j];
		f << "line," << n.w[0] << "," << n.w[1] << "," << n.b << "," << m.output.w[j] << "\n";
	}

	std::cout << "Wrote " << path << "\n";
}

int main()
{
	std::vector<Sample> trainingSamples = createSample(800);
	std::vector<Sample> testingSamples = createSample(200);

	Model m;
	initModel(m);

	train(m, trainingSamples, 10000, 1.0);

	std::cout << accuracy(m, testingSamples) << "\n";

	exportCSV(m, testingSamples, "nn_output.csv");
}