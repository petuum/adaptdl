from adaptdl.conftest import elastic_multiprocessing


@elastic_multiprocessing
def test_allreduce():
    import adaptdl.collective
    import adaptdl.env
    adaptdl.collective.initialize("0.0.0.0")
    # allreduce with default reduce_fn (addition).
    result = adaptdl.collective.allreduce(adaptdl.env.replica_rank())
    assert result == sum(range(adaptdl.env.num_replicas()))
    # allreduce with custom reduce_fn (set union).
    result = adaptdl.collective.allreduce(
        {adaptdl.env.replica_rank()},
        reduce_fn=lambda a, b: a | b)
    assert result == set(range(adaptdl.env.num_replicas()))
    return [5, 0][adaptdl.env.num_restarts()]


@elastic_multiprocessing
def test_allreduce_async():
    import adaptdl.collective
    import adaptdl.env
    adaptdl.collective.initialize("0.0.0.0")
    future_1 = adaptdl.collective.allreduce_async(1)
    future_2 = adaptdl.collective.allreduce_async(2)
    future_3 = adaptdl.collective.allreduce_async(3)
    assert future_2.result() == 2 * adaptdl.env.num_replicas()
    assert future_1.result() == 1 * adaptdl.env.num_replicas()
    assert future_3.result() == 3 * adaptdl.env.num_replicas()
    return [5, 0][adaptdl.env.num_restarts()]


@elastic_multiprocessing
def test_broadcast():
    import adaptdl.collective
    import adaptdl.env
    adaptdl.collective.initialize("0.0.0.0")
    result = adaptdl.collective.broadcast(adaptdl.env.replica_rank())
    assert result == 0
    return [5, 0][adaptdl.env.num_restarts()]
