import imageio
with imageio.get_writer('line.gif', mode='i') as writer:
        image = imageio.imread(f'./savefigs/initial.png')
        writer.append_data(image)
        for i in range(1,63): 
            # print(i,point)
            image = imageio.imread(f'./savefigs/mds-{i}.png')
            writer.append_data(image)
        for i in range(1,326): 
            # print(i,point)
            image = imageio.imread(f'./savefigs/set-mds-{i}.png')
            writer.append_data(image)
       
    