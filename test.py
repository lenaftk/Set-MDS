import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots()

# Preparing dataset
line = [x for x in range(10)]
x = [[5, 2, 4, 8, 5, 6, 8, 7, 1, 3],[0,1],[0,1]]
print(len(x))
text = [str(i%3) for i in range(len(x))]
print(text)
  
# # plotting scatter plot
# plt.scatter(x, y)
  
# # Loop for annotation of all points
# def animate():
#     for i in range(len(x)):
#         plt.annotate(text[i], (x[i],  0.2))
# def animate(i):
#     line.set_xdata(np.sin(x + i / 50))  # update the data.
#     return line,

# ani = animation.FuncAnimation(
#     fig, animate, interval=20, blit=True, save_count=50)
  
# # adjusting the scale of the axes
# plt.xlim((-1, 10))
# plt.ylim((0, 10))
# plt.show()