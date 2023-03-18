import numpy as np
import matplotlib.pyplot as plt
import imageio
np.random.seed(0)
SIZE = 50
y = np.random.randint(-5, 5, SIZE)
plt.plot(y)
plt.ylim(-10, 10)
plt.show()


for i in range(2, SIZE+1):
    plt.plot(y[0:i])
    plt.ylim(-10, 10)
    plt.savefig(f'./savefigs/line-{i}.png')
    plt.close()

with imageio.get_writer('line.gif', mode='i') as writer:
    for i in range(2, SIZE+1):
        image = imageio.imread(f'./savefigs/line-{i}.png')
        writer.append_data(image)
