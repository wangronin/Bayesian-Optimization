from bayes_optim.extension import KernelPCABO

def test_simple():
    c = KernelPCABO.MyOrderedContainer()
    for i in range(10):
        c.add_element(i)
    assert 1 == c.find_pos(0)
    assert 0 == c.find_pos(-1)
    assert 10 == c.find_pos(100)
    assert 2 == c.find_pos(1)
