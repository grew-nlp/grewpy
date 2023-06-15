

class Label(dict):
    def __init__(self, d):
        super().__init__(d)

    def __str__(self):
        return "{" + ",".join([f'"{k}":"{v}"' for k,v in self.items()]) + "}"
    
    def __repr__(self):
        return f"Label({str(self)})"
    

class Observation(object):
    def say_hello(self):
        return f"hello, I am {str(self)} with type {type(self)}"
    
    def __str__(self):
        return str(self.obs)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.obs)})"
    
    @staticmethod
    def from_str(s):
        x = eval(s)
        if isinstance(x, bool):
            return Obool(x)
        if isinstance(x, Label):
            return OLabel(x)
        else:
            raise ValueError ("Unknown observation")
    

class OLabel(Observation):
    def __init__(self, obs):
        if isinstance(obs, Label):
            self.obs = obs
        else:
            raise ValueError (f"OLabel type error: type 'Label' expected but get an expression of type '{type (obs)}'")
        
class Obool(Observation):
    def __init__(self, obs):
        if isinstance(obs, bool):
            self.obs = obs
        else:
            raise ValueError (f"Obool type error: type 'bool' expected but get an expression of type '{type (obs)}'")

if __name__ =="__main__":
    l = Label({"1" : "comp", "2" : "obl", "deep": "at"})

    print(l)
    print(repr(l))
    x = eval(repr(l))
    print(x)

    t = OLabel(x)
    print(t)
    print(repr(t))
    u = eval(repr(t))
    print(u)
    print(repr(u))
    print(u.say_hello())

    z = Observation.from_str('True')
    print(z.say_hello())
    print(str(eval(repr(z))))

