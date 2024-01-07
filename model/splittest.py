s = "a b c? d e f; g h i; j k l. m n"

i = 0
try:
    i = s.index("?")
    print(i)
    s = s[i + 1:].strip()
    try:
        try:
            i = s.rindex(".")
            print(i)
        except:
            i = s.rindex("?")
            print(i)
    except:
        i = s.rindex("!")
        print(i)
except:
    i = s.rindex(".")
    print(i)

if i is not None:
    s = s[0:i + 1].strip()
print(s)
