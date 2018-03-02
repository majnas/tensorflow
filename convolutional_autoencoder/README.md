
### Network Architecture
| Section | Layer Type | Shape |
| --- | --- | --- |
| Encoder | image | 28x28x1 |
| Encoder | conv | 28x28x24 |
| Encoder | pool | 14x14x24 |
| Encoder | conv | 14x14x20 |
| Encoder | pool | 7x7x20 |
| Encoder | conv | 7x7x16 |
| Latent representation | pool | 4x4x16 |
| Decoder | unpool | 7x7x16 |
| Decoder | conv | 7x7x16 |
| Decoder | unpool | 14x14x20 |
| Decoder | conv | 14x14x20 |
| Decoder | unpool | 28x28x20 |
| Decoder | conv | 28x28x20 |
| Decoder | conv | 28x28x1 |

![alt text](https://github.com/m-nasiri/tensorflow/blob/master/convolutional_autoencoder/images/cae.png)
![alt text](https://github.com/m-nasiri/tensorflow/blob/master/convolutional_autoencoder/images/loss_.png)

