import random
import pandas as pd
import argparse

def generate_id(index):
    return f'codeco_pod{index}'

def gen_times(i, last_creation):
    creation = 0
    exec = 0
    
    if i-1 == 0:
        creation = 0
    else:
        # Change range for creation time (minimum of 0), en segons
        creation = last_creation + random.randint(1,20)
    # Change range for exec time (minimum of 1), en segons
    exec = random.randint(5, 100)
    #return creation, exec
    return creation, exec

# code copied from https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that accepts command-line arguments.")
    parser.add_argument("--size", default='100', help="Size of generated dataset")
    parser.add_argument("--output", default='test.csv', help="Default destination file")
    args = parser.parse_args()
    
    
    for data_size in range (0,200):
        #size = int(args.size)
        size = int(data_size)
        filename = args.output.replace(".csv",f"_{human_format(size)}.csv")
        
        dataset = []
        last_creation = 0
        for i in range(1, size + 1):
            id_value = generate_id(i)
            # Change range for cpu cores (minimum of 0.001, maximum can be changed depending of app size)
            cpu = round(random.uniform(1,15), 4)
            if cpu > 1.0:
                real_cpu = round(cpu*random.uniform(0.5,0.99), 4)
            else:
                real_cpu = cpu
            # Change range for ram (GB) (minimum of 1, maximum can be changed depending of app size)
            ram = round(random.uniform(1,20), 4)
            real_ram = round(ram*random.uniform(0.5,0.99), 4)
            creation_time, exec_time = gen_times(i, last_creation)
            last_creation = creation_time
            
            dataset.append([id_value] + [cpu] + [ram] + [creation_time] + [exec_time] + [real_cpu] + [real_ram])
        df = pd.DataFrame(dataset, columns=['name', 'cpu', 'ram', 'creation_time', 'exec_time', 'real_cpu', 'real_ram'])
        # Change path
        df.to_csv(filename, index=False)

        print(f"Succesfully created simulation dataset: {filename}")
