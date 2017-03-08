%% Plotting Performance Curves
totalWaste = 0;
totalWaste_o = 0;
totalWaste_op = 0;
totalWaste_la = 0;
for i=1:NEpisodes
    totalWaste = totalWaste + sum(performance_measures_agent.wastage(i, :));
    wastage_avg(i) = totalWaste / i;
    totalWaste_o = totalWaste_o + sum(performance_measures_oracle.wastage(i, :));
    wastage_avg_o(i) = totalWaste_o / i;
    totalWaste_op = totalWaste_op + sum(performance_measures_oracle2.wastage(i, :));
    wastage_avg_op(i) = totalWaste_op / i;
    totalWaste_la = totalWaste_la + sum(performance_measures_agent_la.wastage(i, :));
    wastage_avg_la(i) = totalWaste_la / i;
end

figure()
plot(1:NEpisodes, wastage_avg);
hold on;
plot(1:NEpisodes, wastage_avg_o);
plot(1:NEpisodes, wastage_avg_op);
xlabel('Episodes','FontSize', 14);
plot(1:NEpisodes, wastage_avg_la);
ylabel('Averaged daily wastage','FontSize', 14);
legend('Agent','Oracle','Oracle2','Linear Agent');

totalbattBen = 0;
totalbattBen_o = 0;
totalbattBen_op = 0;
totalbattBen_la = 0;
for i=1:NEpisodes
    totalbattBen = totalbattBen + sum(performance_measures_agent.bat_charge_cost(i, :));
    bat_benefit_avg(i) = totalbattBen / i;
    totalbattBen_o = totalbattBen_o + sum(performance_measures_oracle.bat_charge_cost(i, :));
    bat_benefit_avg_o(i) = totalbattBen_o / i;
    totalbattBen_op = totalbattBen_op + sum(performance_measures_oracle2.bat_charge_cost(i, :));
    bat_benefit_avg_op(i) = totalbattBen_op / i;
    totalbattBen_la = totalbattBen_la + sum(performance_measures_agent_la.bat_charge_cost(i, :));
    bat_benefit_avg_la(i) = totalbattBen_la / i;
end

% figure()
% plot(1:NEpisodes, bat_benefit_avg);
% % plot(1:NEpisodes, mean(bat_charge_cost,2));
% hold on;
% plot(1:NEpisodes, bat_benefit_avg_o);
% plot(1:NEpisodes, bat_benefit_avg_op);
% plot(1:NEpisodes, bat_benefit_avg_la);
% % plot(1:NEpisodes, mean(bat_charge_cost_o,2));
% xlabel('Episodes','FontSize', 12);
% ylabel('Averaged Savings in the bill due to battery','FontSize', 12);
% legend('Agent','Oracle','Oracle2','Linear Agent');

for i= 1:NBlocks
    sum_0 = 0;
    sum_1 = 0;
    sum_o = 0;
    sum_op = 0;
    sum_2 = 0;
    for j = 1:NEpisodes
        sum_0 = sum_0 + cost_without_battery(j, i);
        bill_without_batt(j, i) = sum_0 / j;
        sum_1 = sum_1 + performance_measures_agent.cost(j, i);
        actual_bill(j, i) = sum_1 / j;
        sum_o = sum_o + performance_measures_oracle.cost(j, i);
        oracle_bill(j, i) = sum_o / j;
        sum_op = sum_op + performance_measures_oracle2.cost(j, i);
        oracle_pred_bill(j, i) = sum_op / j;
        sum_2 = sum_2 + performance_measures_agent_la.cost(j, i);
        actual_bill_la(j, i) = sum_2 / j;
    end
end

figure()
plot(1:NEpisodes,sum(actual_bill,2))
hold on
plot(1:NEpisodes,sum(oracle_bill,2))
plot(1:NEpisodes,sum(oracle_pred_bill,2))
plot(1:NEpisodes,sum(actual_bill_la,2))
% plot(1:NEpisodes,sum(bill_without_batt,2))
xlabel('Episodes','FontSize', 12);
ylabel('Daily bill','FontSize', 12);
title('Average daily power bill');
legend('RL Agent','Oracle with actual data','Oracle with predicted data','Linear Agent');

% figure()
% plot(1:NEpisodes,mean(performance_measures_agent.cost,2))
% hold on
% plot(1:NEpisodes,mean(performance_measures_oracle.cost,2))
% plot(1:NEpisodes,mean(performance_measures_oracle2.cost,2))
% plot(1:NEpisodes,mean(performance_measures_agent_la.cost,2))
% % plot(1:NEpisodes,sum(bill_without_batt,2))
% xlabel('Episodes','FontSize', 12);
% ylabel('Daily bill','FontSize', 12);
% title('Average daily power bill');
% legend('RL Agent','Oracle with actual data','Oracle with predicted data','Linear Agent');