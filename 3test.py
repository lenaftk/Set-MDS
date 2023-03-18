import imageio
with imageio.get_writer('line.gif', mode='i') as writer:
        for i in range(1,22):
            for point in range(3):
                # print(i,point)
                image = imageio.imread(f'./savefigs/1mds-{i}{point}.png')
                writer.append_data(image)
        for k in range(2):
            for turn in range(1,38):
                for point in range(4):
                    # print(i,point)
                    image = imageio.imread(f'./savefigs/mds-{k}-{turn}-{point}.png')
                    writer.append_data(image)
        