import mj_main_opt2 as opt2

params_dic={'Initial':[100.95, 100.95, 656.6, 20.8, 20.8, 15.0, 1.2, 1.2, 1.0, 56.9, 56.9, 25.0, 7.7, 7.7, 9.0, 0.5, 0.5, 0.1,0.0],'DE': [68.82268624911278, 87.30013948660783, 800.9886078885371, 27.56219277367184, 29.386415392317193, 76.47589829741032, 1.2523962173541179, 1.5020288205439072, 9.20760079888003, 46.15611716356331, 66.8605072573911, 51.72971254753558, 4.187172238726845, 4.834878016053025, 4.474852626888598, 0.6850505737992053, 0.6390146059300649, 1.1087607436751954, -0.04533684487772802]}

for method in list(params_dic.keys()):
    r = opt2.obj_fun(params_dic[method], render=True)
    print ('settling time (%s):'%method,r[11])
    opt2.error_plot(r,method)
    #opt2.simple_plot(r)
    opt2.total_plot(r,method)
