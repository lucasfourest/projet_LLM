

class PVP:

    def __init__(self,id_p,id_v):
        self.pattern=Pattern(id_p)
        self.verbalizer=Verbalizer(id_v)
        

    def transform_data(self,x,y):
        pattern=self.pattern.transform_input(x)
        verb=self.verbalizer.transform_label(y)
        return pattern,verb


class Pattern:

    def __init__(self,id):
        self.id=id
        if self.id==1:
            self.prefix='the review is: '
            self.suffix=' Is it a positive review?'
        if self.id==2:
            self.prefix=''
            self.suffix=' Did this user liked the movie?'
        

    def transform_input(self,x):
        return self.prefix+x+self.suffix


class Verbalizer:

    def __init__(self,id):
        self.id=id
        if self.id==1:
            self.answer_pos='yes'
            self.answer_neg='no'

        

    def transform_label(self,y):
        if y==1:
            return self.answer_pos
        return self.answer_neg



