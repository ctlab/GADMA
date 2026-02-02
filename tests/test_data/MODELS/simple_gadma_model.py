import gadma

def model_func():
    nu1 = gadma.utils.PopulationSizeVariable("nu1")
    t1 = gadma.utils.PopulationSizeVariable("t1")

    model = gadma.models.EpochDemographicModel()
    model.add_epoch(
        time_arg=t1,
        size_args=[nu1],
        mig_args=None,
        dyn_args=None,
    )
    return model
