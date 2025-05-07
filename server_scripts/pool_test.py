from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == '__main__':
    task_list = [1, 3, 2, 4]
    with Pool(processes=4) as pool:
        results = pool.map(square, task_list)

    print(results)
