import bmtrain as bmt

def main():
    bmt.init_distributed()
    bmt.print_rank("======= Send Recv =======")
    bmt.benchmark.send_recv()
    

if __name__ == '__main__':
    main()
