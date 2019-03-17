# FCN-pytorch-CanRun
FCN-Pytorch FOR beginners

FOR ALL semantic segmentation beginners! This repo is modified from https://github.com/pochih/FCN-pytorch, thanks this lovely bro.

Enviroment: Pytorch 1.0.0 python 3.7 cuda10

It's a east implemention of FCN using pytorch，Just need change the datasets path.

If you want to use Camvid dataset,in CamVid file,you should:

python CamVid_utlis.py
python train.py
-----if you stop the trainning half time,you can run :python train.py --resume checkpoint.tar To resuming the train
If you want to use Cityscapes dataset,in Cityscapes file,you should:

python Cityscapes_utlis.py
python train.py
-----if you stop the trainning half time,you can run :python train.py --resume checkpoint.tar To resuming the train
If you have some problems,welcome to ISSUE or send e-mail to me. Thanks

个人矫情，回顾一下小白入门的辛酸泪

为了完成研究生毕业，不得不放下这块了。 几乎每一行都注释了，完全的新手入门啊。 回想之前四处求医的日子，很多感触，感谢一路上帮助我的陌生人，感谢曾经回我邮件的朋友们， 特别感谢一位浙江理工的研三学长，也特别感谢一位武大做医疗图像的学姐，在并没有义务的情况下，给了我莫大的帮助与鼓励 我只有说好人一生平安一生运气爆棚一生幸福了。 希望以后做这方面的同学们能挺过入门时候的艰辛（大神请无视掉），好好坚持，毕竟这方向真的挺好

同一个老师门下，老师收两个专业，要是专业能够互换那该多好，谁让我是调剂到一个没法做图像的专业呢？ 以此纪念我的第一个语义分割demo，但我不希望是最后一个。 未来，加油，祝好
