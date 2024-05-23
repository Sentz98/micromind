import os
import torch
import torch.ao.quantization as tq

def find_layers_in_order(model, layer_types):
    layer_names = []
    modules = list(model.named_modules())

    for i in range(len(modules)):
        block = []
        name, module = modules[i]

        if type(module) == layer_types[0]:
            block.append(name)

            for j in range(1, len(layer_types)):
              if i + 1 < len(modules):
                  name, module = modules[i + j]
                  if type(module) == layer_types[j]:
                    block.append(name)

        if len(block) == len(layer_types):
          layer_names.append(block)
          
    return layer_names

def fuse_modules(model, layers_types, is_qat = False, inplace=False):
    layer2fuse = find_layers_in_order(model, layers_types)

    if not is_qat:
      model.eval()

    fuse_func = tq.fuse_modules_qat if is_qat else tq.fuse_modules
    model_fused = fuse_func(model, layer2fuse, inplace=False)

    if not inplace:
      return model_fused
    
def inject_quant(model):
    def forward_injected(self, x):
        x = self.quant(x)
        x = original_forward(self,x)
        x= self.dequant(x)
        return x
    
    # add quant and dequant layers
    model.quant = tq.QuantStub()
    model.dequant = tq.DeQuantStub()
    
    # Bind the new method to the instance
    original_forward = model.__class__.forward
    bound_forward = forward_injected.__get__(model, model.__class__)
    setattr(model.__class__, "forward", bound_forward)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size of the model(MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def log_results(model, test_loss, test_acc_1, inference_time, test_acc_5 = None):
    print("\n========================================= PERFORMANCE =============================================")
    print_size_of_model(model)
    log_str = f'\nTest Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1:6.2f}%'
    if test_acc_5:
        log_str += f'| Test Acc @5: {test_acc_5:6.2f}%'
    print(log_str)
    print('Average inference time = {:0.4f} milliseconds'.format(inference_time))
    print("====================================================================================================")
