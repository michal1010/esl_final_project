# Steps to Run
- Install the requirements given in requirements.txt
- Copy the directory "federated_data_random" to directory where you have your federated model files
- execute flwr run . in terminal, **ensure that the working directory is the one where you have all the fedarated model files before executing the command**
- You can change pyproject.toml attributes such as num-server-rounds (represents the number of rounds the fedarated setup runs basically **train -> send params to server -> aggregate paramas -> client updates to aggregated params -> Warm start with aggregated params**), and options.num-supernodes (the number of clients)
- **Useful Resources:** https://flower.ai/docs/ , https://github.com/adap/flower