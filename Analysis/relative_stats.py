import sys
import math

if __name__ == "__main__":
	log_name = sys.argv[1]
	f = open(log_name, 'r')
	text = f.read()
	exec('data =' + text)

	accumulator = 0.0
	accumulatorSqrd = 0.0
	n_lost = 0
	for game in data:
		difference = game[0] - max([game[1],game[2]])
		if difference <= 0:
			n_lost += 1
		accumulator += difference
		accumulatorSqrd += difference*difference
	mean = float(accumulator)/len(data)
	variance = float(accumulatorSqrd)/len(data) - mean*mean
	print(mean)
	print(math.sqrt(variance))
	print(n_lost)