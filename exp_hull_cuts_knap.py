import exp_hull_cuts as ehc
import knapsack_loader as kl

def main():
    import plot_utils as pu

    wants_plots = False

    for model in kl.generate(1, 2, 20, 5, 10, 1000, equality=False, seed=42):
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
        ehc.run_cuts(model, rounds=550, verbose=False, callback=(plotter_callback if wants_plots else None))
        if plotter is not None:
            plotter.render()

if __name__ == "__main__":
    main()