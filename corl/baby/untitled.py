
with open('./test_pkl.csv', 'a+') as file:
    head = 'Itr,'
    for meta_batch in range(4):
        meta_head = 'Task{},{},{},{},'.format(meta_batch, 'AverageDiscountedReturn', 'UndiscountedReturn', 'SuccessRate')
        head += meta_head
    head = head[:-1] + '\n'
    print(head)
    file.write(head)
    import ipdb
    ipdb.set_trace()
    print('...')
