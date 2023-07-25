hostnames = [
    "ip-172-31-31-108",
    "ip-172-31-21-126",
    "ip-172-31-25-208",
    "ip-172-31-18-32",
    "ip-172-31-18-199",
    "ip-172-31-31-120",
    "ip-172-31-26-219",
    "ip-172-31-17-75",
    "ip-172-31-33-200",
    "ip-172-31-45-68",
    "ip-172-31-34-125",
    "ip-172-31-32-250",
    "ip-172-31-34-66",
    "ip-172-31-33-127",
    "ip-172-31-35-212",
    "ip-172-31-45-206",
    "ip-172-31-42-86",
    "ip-172-31-34-85",
    "ip-172-31-45-203",
    "ip-172-31-42-73",
    "ip-172-31-44-205",
    "ip-172-31-44-203",
    "ip-172-31-32-158",
    "ip-172-31-41-156",
    "ip-172-31-46-8",
    "ip-172-31-35-132",
    "ip-172-31-35-139",
    "ip-172-31-34-138",
    "ip-172-31-34-60",
    "ip-172-31-46-59",
    "ip-172-31-45-0",
    "ip-172-31-35-128",
    "ip-172-31-38-22",
    "ip-172-31-39-21",
    "ip-172-31-42-152",
    "ip-172-31-35-24",
    "ip-172-31-44-15",
    "ip-172-31-41-140",
    "ip-172-31-38-20",
    "ip-172-31-37-16",
]

# shuffle_ranks = [0, 42, 36, 19, 15, 13, 55, 11, 1, 10, 43, 45, 39, 12, 63, 5, 37, 59, 35, 31, 20, 27, 17, 28, 41, 3,
#                 62, 21, 47, 32, 22, 51, 46, 2, 9, 44, 16, 61, 30, 52, 8, 50, 58, 57, 25, 6, 40, 14, 49, 48, 18, 54,
#                 33, 38, 23, 60, 53, 4, 29, 34, 56, 7, 26, 24]
shuffle_ranks = [0, 31, 6, 4, 34, 11, 18, 5, 39, 14, 7, 16, 21, 23, 25, 29, 37, 3, 30, 38, 26, 10, 12, 22, 15, 13, 24,
                 1, 2, 9, 32, 19, 33, 36, 8, 17, 20, 28, 27, 35]


def shuffle_hostnames_case2():
    assert (len(hostnames) == 40)
    with open("./ds_hostnames_shuffled", 'w') as output:
        for rank in shuffle_ranks:
            if rank < 8:
                output.write(hostnames[rank] + ' slots=4\n')
            else:
                output.write(hostnames[rank] + ' slots=1\n')


def no_shuffle_hostnames_case2():
    assert (len(hostnames) == 40)
    with open("./ds_hostnames_shuffled", 'w') as output:
        for rank in range(40):
            if rank < 8:
                output.write(hostnames[rank] + ' slots=4\n')
            else:
                output.write(hostnames[rank] + ' slots=1\n')


def shuffle_hostnames_case345():
    assert(len(hostnames) == 64)
    with open("./ds_hostnames_shuffled", 'w') as output:
        for rank in shuffle_ranks:
            output.write(hostnames[rank] + ' slots=1\n')


# shuffle_hostnames_case345()
shuffle_hostnames_case2()
# no_shuffle_hostnames_case2()