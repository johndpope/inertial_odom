p=[0;0;0;1];
g=[0;0;0;1];
figure(1)
n=size(pred_r)
for i=1:n(1)
    T_p=pred_r(i,:,:);
    T_p=reshape(T,[4,4]);
    p=T*p;
    plot3(p(1),p(2),p(3),'g*-')
    hold on
end
figure(2)
for i=1:n(1)
    T_g=gt_r(i,:,:);
    T_g=reshape(T,[4,4]);
    g=T_g*g;
    plot3(g(1),g(2),g(3),'b*-')
    hold on
end