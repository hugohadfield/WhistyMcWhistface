import sys
import math

if __name__ == "__main__":
	log_name = sys.argv[1]
	f = open(log_name, 'r')
	text = f.read()
	exec('data =' + text)

	sums = [0.0 for i in data[0]]
	sumsSqrd = [0.0 for i in data[0]]
	for game in data:
		for number,score in enumerate(game):
			sums[number] += score
			sumsSqrd[number] += score*score
	averages = [float(a)/len(data) for a in sums]
	EX2 = [float(a)/len(data) for a in sumsSqrd]
	std_devs = [ math.sqrt(EX2[i] - averages[i]*averages[i]) for i in range(len(averages))]
	print(averages)
	print(std_devs)
	std_errs = [ i/math.sqrt(len(data)) for i in std_devs]
	print(std_errs)
