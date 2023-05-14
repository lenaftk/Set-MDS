import imageio
with imageio.get_writer('line.gif', mode='i') as writer:
        image = imageio.imread(f'./savefigs/initial.png')
        writer.append_data(image)
        for i in range(1,132): 
            # print(i,point)
            image = imageio.imread(f'./savefigs/mds-{i}.png')
            writer.append_data(image)
        for i in range(1,401): 
            # print(i,point)
            image = imageio.imread(f'./savefigs/set-mds-{i}.png')
            writer.append_data(image)
       
    