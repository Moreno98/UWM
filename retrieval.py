import utils.arg_parse as arg_parse

def main(opt):
    retrieval_handler = opt['retrieval_handler'](opt)
    retrieval_handler.run()    

if __name__ == '__main__':
    opt = arg_parse.retrieval()
    main(opt)