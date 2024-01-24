

class PVP:

    def __init__(self,id_p,id_v,dataset='imdb'):
        self.pattern=Pattern(id_p,dataset=dataset)
        self.verbalizer=Verbalizer(id_v,dataset=dataset)
        

    def transform_data(self,inputs,y):
        pattern=self.pattern.transform_input(inputs)
        verb=self.verbalizer.transform_label(y)
        return pattern,verb


class Pattern:

    def __init__(self,id,dataset='imdb'):
        self.id=id
        self.dataset=dataset

        if self.dataset=='imdb':
            if self.id==1:
                self.prefix='The review is: '
                self.suffix=' Is it a positive review?'
            if self.id==2:
                self.prefix=''
                self.suffix=' Did this user like this movie?'
            if self.id==3:
                self.prefix='Read the following review: '
                self.suffix=' Did this user enjoy its experience?'
            # if self.id==4:
            #     self.prefix='the review is: "'
            #     self.suffix='". Is it a positive review?'
            # if self.id==5:
            #     self.prefix='"'
            #     self.suffix='". Did this user like this movie?'
            # if self.id==6:
            #     self.prefix='Read the following review: "'
            #     self.suffix='". Did this user enjoy its experience?'
        
        if self.dataset=='boolq':

            if self.id==1:
                self.prefix=' Question: '
                self.suffix='? Answer:'

            if self.id==2:
                self.prefix=' Based on the previous passage, '
                self.suffix='?'
            


    def transform_input(self,inputs):
        if self.dataset=='imdb':
            x=inputs[0]
            return [self.prefix+x+self.suffix,'']
        if self.dataset=='boolq':
            if self.id==1 or self.id==2:
                p,q=inputs

                return [p+self.prefix+q+self.suffix,'']


class Verbalizer:

    def __init__(self,id,dataset='imdb'):
        self.id=id
        if self.id==1:
            self.answer_pos='yes'
            self.answer_neg='no'

        

    def transform_label(self,y):
        if y==1:
            return self.answer_pos
        return self.answer_neg



