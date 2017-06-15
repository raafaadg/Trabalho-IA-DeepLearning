function [er, bad, a3, tax_er] = cnntest(net, x, y)
    %  feedforward
    net = cnnff(net, x);
    [~, h] = max(net.o);
    [~, a] = max(y);
    bad = find(h ~= a);
    aux=h(bad);
    [a1,b1]=hist(a,unique(a));
    [a2,b2]=hist(aux,unique(aux));
    a3=1-((a1-a2)./a1);
    tax_er=1-((numel(a)-numel(bad))/numel(a));
    er = numel(bad) / size(y, 2);
    
    figure;
    title('20 Primeiros Erros')
    for i = 1:20
        subplot(4,5,i);
        imshow(x(:,:,bad(i)))

        [position,aux]  = find(y(:,bad(i))>0);
        label(i)        = position - 1;
        title(['Label: ' num2str(label(i))]);
    end

    fprintf('Mean Squared Error (MSE) = %f\n\n',er)
    fprintf('Taxa de Erro para valores de 0 a 9 = [%f %f %f %f %f %f %f %f %f %f]\n\n',a3')
end
