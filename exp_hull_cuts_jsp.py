import exp_hull_cuts as ehc
import jsplib_loader as jl

def main():
    import plot_utils as pu

    wants_plots = False
    instances = jl.get_instances()

    for instance in [instances["abz4"]]:
        model = instance.as_gurobi_balas_model(use_big_m=True)
        print(f"Instance {instance.name}, constraints {model.NumConstrs}, known optimum {instance.score}")
        plotter: pu.PlotterBase = None
        last_count = model.NumConstrs

        def plotter_callback(mdl):
            nonlocal last_count, plotter
            if mdl.NumConstrs > last_count and plotter is not None:
                for c in mdl.getConstrs()[last_count:]:
                    plotter.add_constraint(c)
            else:
                plotter = pu.create(mdl)
                if plotter is not None:
                    plotter.add_ball(1.5)
        ehc.run_cuts(
            model,
            rounds=150,
            verbose=False,
            callback=(plotter_callback if wants_plots else None),
            known_opt_obj=instance.score,
            debug_track_invalid=True,
        )
        if plotter is not None:
            plotter.render()

if __name__ == "__main__":
    main()