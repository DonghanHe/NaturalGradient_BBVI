import torch
from itertools import chain
from tqdm import trange

def train_model_a(model_type,
                  model_hyper_parameters,
                  trainloader,
                  evalloader,
                  device,
                  epochs = 100,
                  natgrad = False,
                  optimizer = None,
                  lr = 1e-3,
                  ):
    model = model_type(**model_hyper_parameters, natgrad=natgrad).to(device)

    for e in epochs:
        # TODO: write train loop
        for batch_idx, data in enumerate(trainloader):
            # TODO: write per step loop
            pass
    pass

def train_model_na(model_type,
                   model_hyper_parameters,
                   variational_parameter_shape,
                   trainloader,
                   evalloader,
                   eval_metric,
                   device,
                   variational_parameter_init = torch.nn.init.xavier_uniform_,
                   epochs = 100,
                   eval_every = 10,
                   natgrad = False,
                   optimizer = None,
                   optimizer2 = None,
                   lr = 1e-3,
                   lr2 = 1e-3,
                   opt2_clip = None,
                   verbose = True,
                   ):
    model = model_type(**model_hyper_parameters, natgrad=natgrad).to(device)
    variational_parameters = torch.nn.Parameter(torch.FloatTensor(*variational_parameter_shape),)
    variational_parameter_init(variational_parameters)
    variational_parameters.to(device)
    if optimizer2 is None:
        opt = optimizer(chain(model.parameters(), [variational_parameters,]), lr=lr)
    else:
        opt2 = optimizer([variational_parameters,], lr=lr)
        opt = optimizer2(model.parameters(), lr=lr2)

    tot_loss_rec = []
    nll_rec = []
    kl_z_rec = []
    metric_rec = []
    rg = trange(epochs) if verbose else range(epochs)
    if evalloader is None or eval_metric is None:
        eval_every = epochs+1

    torch.autograd.set_detect_anomaly(True)
    for e in rg:
        epoch_nll = 0.
        epoch_kl_z = 0.
        count = 0

        model.train()
        for data, idx in trainloader:
            z_logits_i = variational_parameters[idx, :]

            opt.zero_grad()

            if type(data) is list:
                data = (d.to(device) for d in data)
                # this is for training LDA
            else:
                data = (data.to(device),)

            logp, kl_z = model(*data, z_logits_i)

            epoch_nll -= logp.item()
            epoch_kl_z += kl_z.item()

            loss = -logp + kl_z

            # if torch.isnan(loss):
            #     print('loss became nan first')
            #     for i, term in enumerate([logp, kl_d, kl_t]):
            #         if torch.any(torch.isnan(term)):
            #             print("term {}".format(i))
            #             if i == 2:
            #                 vp = torch.nn.Softplus()(variational_parameters)
            #                 print(vp.max(), vp.min())

            loss.backward()
            if optimizer2 is None:
                opt.step()
            else:
                if opt2_clip is not None:
                    torch.nn.utils.clip_grad_norm_([variational_parameters,], opt2_clip)
                opt.step()
                opt2.step()
            count += 1

        nll_rec.append(epoch_nll/count)
        kl_z_rec.append(epoch_kl_z/count)
        tot_loss_rec.append((epoch_nll + epoch_kl_z)/count)

        # eval every eval_every epochs
        if (e+1) % eval_every== 0:
            model.eval()
            with torch.no_grad():
                metric_loss = eval_metric(evalloader, model, device)
                metric_rec.append(metric_loss)

    results= dict()
    results['model'] = model
    results['var_params'] = variational_parameters
    results['loss_records'] = {'nll': nll_rec,
                               'kl': kl_z_rec,
                               '-elbo': tot_loss_rec}
    results['metric_record'] = metric_rec

    return results
